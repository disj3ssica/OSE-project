### Figure 2.1###
def mafia_presence():
    df1 = data[data['year'] >= 1983]
    df2 = df1.groupby(['region', 'reg'])[['gdppercap', 'mafia', 'murd', 'ext', 'fire', 'kidnap', 'rob', 'smug',
                                      'drug', 'theft', 'orgcrime']].mean()
    df2 = df2.reset_index()
    
    color = np.where((df2['reg'] == 15) | (df2['reg'] == 18) | (df2['reg'] == 19), 'midnightblue',           # EXCLUDED
                 np.where((df2['reg'] == 16) | (df2['reg'] == 17), 'mediumslateblue',                    # TREATED
                 np.where((df2['reg'] <= 12) | (df2['reg'] == 20), 'salmon', 'none')))                   # THE REST

    df2.plot.scatter('mafia', 'gdppercap', c = color, s = 10, linewidth = 3, 
                 xlabel = 'Presence of mafia organisations', ylabel = 'GDP per capita', ylim = [7000,15000], xlim = [0,2.25],
                 title = 'Figure 2.1: GDP per capita and mafia presence, 1983–2007 average')
    n = ['Basilicata', 'Calabria', 'Campania', 'Apulia', 'Sicily']
    j, z = 0, [1, 2, 3, 16, 18]
    for i in z:
        plt.annotate(n[j], (df2.mafia[i], df2.gdppercap[i]), xytext = (0,1), 
                 textcoords = 'offset points', ha = 'left', va = 'bottom', rotation = 15)
        j += 1

